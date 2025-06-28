import argparse
from preprocessor import ImagePreprocessor
from pin_manager import PinManager
from art_generator import StringArtGenerator
from preview import PreviewRenderer

def main():
    parser = argparse.ArgumentParser(description='Generador de String Art')
    parser.add_argument('input_image', help='Ruta de la imagen de entrada')
    parser.add_argument('--size',     type=int, default=500,
                        help='Tamaño del lienzo (px)')
    parser.add_argument('--pins',     type=int, default=300,
                        help='Número de pines')
    parser.add_argument('--lines',    type=int, default=4000,
                        help='Número de hilos (iteraciones)')
    parser.add_argument('--start-pin',type=int, default=0,
                        help='Índice del pin inicial')
    parser.add_argument('--output',   default='preview.png',
                        help='Ruta de salida de la preview')
    args = parser.parse_args()

    # 1. Preprocesar
    prep = ImagePreprocessor(args.size)
    img = prep.load_and_resize(args.input_image)
    target = prep.to_gray_array(img)

    # 2. Pines
    pm = PinManager(center=(args.size/2, args.size/2),
                    radius=args.size/2,
                    num_pins=args.pins)
    pins = pm.get_pins()

    # 3. Generar string art
    gen = StringArtGenerator(target=target,
                             pins=pins,
                             iterations=args.lines,
                             gamma=0.9)
    seq = gen.generate(start_pin=args.start_pin)

    # 4. Render y guardado
    renderer = PreviewRenderer(size=args.size)
    preview = renderer.render(pins, seq)
    preview.save(args.output)
    print(f'Preview guardada en {args.output}')

if __name__ == '__main__':
    main()
